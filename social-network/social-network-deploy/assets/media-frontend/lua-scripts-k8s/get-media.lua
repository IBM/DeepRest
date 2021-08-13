local _M = {}

local function _StrIsEmpty(s)
  return s == nil or s == ''
end

local function _StringSplit(input_str, sep)
  if sep == nil then
    sep = "%s"
  end
  local t = {}
  for str in string.gmatch(input_str, "([^"..sep.."]+)") do
    table.insert(t, str)
  end
  return t
end

function _M.GetMedia()
  local bridge_tracer = require "opentracing_bridge_tracer"
  local mongo = require "resty-mongol"
  local ngx = ngx

  local tracer = bridge_tracer.new_from_global()
  local parent_span_context = tracer:binary_extract(ngx.var.opentracing_binary_context)
  local span = tracer:start_span("MongoGetMedia", {["references"] = {{"child_of", parent_span_context}}})
  local carrier = {}
  tracer:text_map_inject(span:context(), carrier)

  local chunk_size = 255 * 1024

  ngx.req.read_body()
  local args = ngx.req.get_uri_args()
  if (_StrIsEmpty(args.filename)) then
    ngx.header.content_type = "text/plain"  
    ngx.status = ngx.HTTP_BAD_REQUEST
    ngx.say("Incomplete arguments")
    ngx.log(ngx.ERR, "Incomplete arguments")
    span:finish()
    ngx.exit(ngx.HTTP_BAD_REQUEST)
  end


  local conn = mongo()
  conn:set_timeout(5000)
  local ok, err = conn:connect("media-mongodb.social-network.svc.cluster.local", 27017)
  if not ok then
    ngx.log(ngx.ERR, "mongodb connect failed: "..err)
  end
  local db = conn:new_db_handle("media")
  local col = db:get_col("media")

  local media = col:find_one({filename=args.filename})
  if not media then
    ngx.header.content_type = "text/plain" 
    ngx.status = ngx.HTTP_INTERNAL_SERVER_ERROR
    ngx.log(ngx.ERR, "mongodb failed to find: ".. args.filename)
    ngx.say("mongodb failed to find: ".. args.filename)
    span:finish()
    ngx.exit(ngx.HTTP_INTERNAL_SERVER_ERROR)
  end

  local media_file = media.file

  local filename_list = _StringSplit(args.filename, '.')
  local media_type = filename_list[#filename_list]

  if (media_type == "mp4") then
    ngx.header.content_type = "video/" .. media_type
  else
    ngx.header.content_type = "image/" .. media_type
  end
  ngx.say(media_file)

  span:finish()
end

return _M